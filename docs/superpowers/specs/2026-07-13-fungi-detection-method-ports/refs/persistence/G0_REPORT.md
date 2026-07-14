# A11 G0 candidate report

**Status:** ready for independent review; not yet approved.

## Gate results

| Requirement | Result | Evidence |
|---|---|---|
| Exact executable source | PASS | official GUDHI 3.13.0 tag `5dbe510...`, selected local source, official local wheel |
| Provenance and license | PASS | `PROVENANCE.json`, `ATTRIBUTION.md`, `upstream/LICENSE:1-21` |
| Primary theory | PASS | author-hosted paper PDF/text plus ISTA DOI metadata |
| Reconstruction decision | PASS | no reconstruction source found in selected corpus; analysis-only Path 1 selected |
| Frozen contract | PASS | corrected A11 plan plus `RECONCILIATION.md` and `DRIFT.md` |
| Runnable source oracle | PASS | eight controlled cases; exact fixture SHA-256 `5d8c0a17...` |
| Independent logic oracle | PASS | explicit cubical boundary reduction and independent Betti curves |
| Plateau policy | PASS | exact cells are pinned-version drift evidence; intervals/topology decide fidelity |
| Required mutants | PLANNED | named one-test-per-mutant matrix; execution begins at G1 |
| Production code | ABSENT | correctly blocked pending independent G0 PASS |

## Numerical controls

- Four-peak superlevel landscape has finite beta-0 lifetimes 1, 2, and 3, one essential
  beta-0 class, and one beta-1 lifetime 1.
- `min_persistence=2` excludes the lifetime-2 class and retains lifetime 3, proving strict
  threshold equality.
- A sublevel ring has exactly one beta-1 interval `[0, 2)`.
- Diagonally touching top cells form one 8-connected beta-0 class.
- Single-cell sublevel/superlevel controls prove essential death signs `+inf` and `-inf`.
- A non-square input proves Fortran-flat pair IDs convert to `(row, column)` without axis swaps.

## Important source distinction

The foundational paper uses pixels as vertices and 4-connectivity. This A11 contract deliberately
uses GUDHI top-dimensional cells and the matching coface-pair API, producing shared-corner
8-connectivity. This is an explicit drift row, not an unnoticed convention mix.

## Review request

The reviewer must independently read the local paper and selected source, rerun the fixture twice,
run the standalone logic script, verify all cited line ranges and checksums, audit the analysis-only
scope decision, and return explicit PASS or FAIL on the exact commit. Any semantic correction
requires a successor commit and renewed review.
