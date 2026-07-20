# Fungi method ports release evidence

**Evidence date:** 2026-07-13
**Semantic review baseline:** `0f41118f3f7920529ce4fadd55d15e52cfb6bdfe`
**Release scope:** eight implemented paths, three deferred paths

This manifest records the evidence required by `99-integration-and-release.md`. It establishes
source-contract fidelity and integration safety for the implemented scope. It does not establish
improved fungal detection, biological validity, or runtime performance. Those claims require a
separate ground-truth benchmark.

## Disposition by cluster

| Cluster | Release disposition | Core or final implementation commit | Independent review disposition |
|---|---|---|---|
| A01 GWDT | Implemented core plus opt-in APP2 detector seam | `82fcb1318`, seam finalized by `0f41118f3` | G6 PASS on `0f41118f3` |
| A02 tensor voting | Implemented pure numerical core; detector use deferred by the frozen contract | `dcf64b818` | G6 PASS on `b38bff56a` |
| A03 Jerman | Deferred at G0; authenticated paper unavailable after network loss | none | No implementation sign-off claimed |
| A04 bowler-hat | Deferred at G0; MATLAB oracle unavailable and source-license/formula fork unresolved | none | No implementation sign-off claimed |
| A05 Kalman | Deferred at G0; no executable source matches the proposed image-space tracker | none | G0 deferral independently reviewed |
| A06 cellular automaton | Implemented narrow TrickTrack cellular-automaton core | `e1e5cefd0` | G6 PASS on `b38bff56a` |
| A07 NFA | Implemented clean-room binomial NFA statistic only | `a121ddced` | G6 PASS on `b38bff56a` |
| A08 RORPO | Implemented clean-room four-direction 2-D core | `694bde067` | G6 PASS on `0f41118f3` |
| A09 rolling Hough | Implemented Clark rolling-Hough core; coherence remains outside scope | `47429c59c` | G6 PASS on `b38bff56a` |
| A10 FilFinder | Implemented lazy optional `FilFinderDetector` adapter | `a166c79e7` | G6 PASS on `b38bff56a` |
| A11 persistence | Implemented GUDHI cubical-persistence analysis path only | `baf228cfb` | G6 PASS on `b38bff56a` |

The distribution-safe source authority, immutable revisions, hashes, licenses, executable
commands, reconciliation, fixtures, drift rows, and mutation matrices live under
`docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/`. A07 and A08 are clean-room
exceptions: HEAD contains only their sanitized implementation-side artifacts. Complete
oracle-side evidence is isolated at commits `62406d7c2` for A07 and `dd0c966aa` for A08 and was
available to their independent reviewers, not their clean-room implementers.

## Shared integration

The serialized integration commits are:

- `be2d06428`: reviewed public exports, optional topology dependencies, license material, and
  import/public tests.
- `b38bff56a`: A01 APP2 detector seam, package-license correction, attribution correction,
  seam mutations, and exact reconnect export checks.
- `3769381cb`: APP2 threshold/gap, seed/tie, tile/halo/slicing/ownership drift controls.
- `fb4563801`: exact logic-script manifest, shared closed-set aliases, and optional-dependency-free
  FilFinder pipeline serialization.
- `0f41118f3`: exact Vaa3D neighbor-order drift mutant and path-visible counterexample.

The legacy `reconnect_strategy="dijkstra"` remains the default. APP2 is opt-in. FilFinder and
GUDHI imports remain lazy and are pinned only in the `topology` extra. Packaging includes the
reviewed third-party license files and excludes reference corpora, fixtures, oracles, and optional
dependency source code.

## Numerical and regression evidence

| Gate | Result |
|---|---|
| Focused combined suite | 518 passed on evidence commit `2bb5c49c9`; no production delta followed the semantic baseline |
| Fresh G7 reviewer suite | 352 passed, 4 deselected; 291 shared registry/serialization/legacy tests passed |
| Broader reviewer suite | 860 passed, 3 skipped before requested finalization |
| Standalone numerical validators | Exact manifest of 8 scripts; all 8 passed |
| Packaging | 4 wheel/sdist tests passed; package-exclusion checks passed |
| A01 seam after final S09 correction | 15 focused tests passed; 12/12 seam mutants killed |
| New-file typing | Targeted mypy passed for the 10 new reconnect, FilFinder, and persistence source files |
| Formatting/lint | Targeted Ruff passed |

The exact script manifest is enforced by `tests/unit/test_fungi_port_public_api.py`; removing or
adding a validator without updating the approved manifest fails the suite. The scripts are under
`docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/` and depend
only on the allowed numerical stack.

## Mutation evidence

| Scope | Individually killed mutants |
|---|---:|
| A01 GWDT core | 16/16 |
| A01 detector seam | 12/12 |
| A02 tensor voting | 15/15 |
| A06 cellular automaton | 12/12 |
| A07 NFA | 13/13 |
| A08 RORPO | 19/19 |
| A09 rolling Hough | 18/18 |
| A10 FilFinder | 21/21 |
| A11 persistence | 15/15 |

The A07 named 13-mutant matrix is the distribution-safe executable tuple in
`tests/unit/sdk_/reconnect/run_nfa_mutations.py`. A08's complete documented oracle-side matrix
remains isolated at `dd0c966aa`. Every other distribution-safe HEAD matrix names its killing test.
The A01 seam runner also verifies that its documented matrix and executable mutant names are
identical.

## Review gates and remaining release decisions

- G0-G6 are complete for A01, A02, and A06-A11 within their explicitly narrowed contracts.
- A01 G6 passed on `0f41118f3` after the returning reviewer independently verified the exact
  source-order mutant and every earlier seam finding.
- The fresh combined reviewer passed G7 code on `2bb5c49c9`; its remaining findings concerned
  this evidence ledger and the phase-gate wording only.
- A03-A05 are not implemented and are not represented as completed algorithms. Their algorithm
  plans record the evidence needed to resume them.
- Public distribution still requires human approval of A02's conflicting upstream license claims
  and the A07 clean-room distribution boundary. These are legal/release decisions, not numerical
  pass results.
- Full-repository mypy and Ruff contain pre-existing failures outside the new numerical modules.
  Targeted changed-file gates pass; this manifest does not relabel the repository-wide baseline
  as green.

G8 public release remains blocked by the two human licensing decisions.
