# SubtractPolySurface golden fixture — attribution

`subtract_poly_surface_gwyddion.npz` holds numeric outputs of Gwyddion 2.71
(Nečas & Klapetek, *Cent. Eur. J. Phys.* 10(1):181–188, 2012, doi:10.2478/s11534-011-0096-2),
GPL-2.0-or-later, computed by a harness that links Gwyddion's libgwyddion/libprocess on
deterministic inputs from `docs/superpowers/plans/2026-10-08-subtract-poly-surface/oracle_inputs.py`.

Only numbers are in this repository. No Gwyddion source, and not the harness (which calls
Gwyddion's API), is distributed, imported, linked, copied, or transcribed by PhenoTypic.
Provenance (tarball sha256, harness sha256, build) is in the sibling `.json`.

## Clean-room record

The production modules

- `src/phenotypic/enhance/_poly_surface_kernels.py` (the float64 kernels), and
- `src/phenotypic/enhance/_subtract_poly_surface.py` (the `SubtractPolySurface` operation)

were written by implementers who did not open the Gwyddion source or the Gwyddion user guide.
Everything they received:

- `docs/superpowers/specs/2026-10-08-subtract-poly-surface/design.md` — the source-free
  behavioural contract (§4).
- `docs/superpowers/specs/2026-10-08-subtract-poly-surface/drift-register.md` — deviations, as
  behaviour statements.
- `docs/superpowers/plans/2026-10-08-subtract-poly-surface/plan.md` — task text with exact
  interfaces (signatures, spec sections, test names) and **complete test code**, but no
  implementation bodies. Its author had read the Gwyddion source; the plan states behaviour and
  tests behaviour, which is within the precedent cited below.
- Per-task briefs, the plan's global constraints, and the review reports and fix briefs that
  followed each task (session working files, not committed). They restate the documents above and
  the reviewers' measurements; none carries Gwyddion source text.
- The logic-validation scripts under
  `docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/` (stdlib + numpy +
  scipy; they never import Gwyddion or PhenoTypic).
- This fixture (numbers only).

`docs/superpowers/specs/2026-10-08-subtract-poly-surface/references.md` was reachable through
`design.md`'s links. It records behaviour facts with `file:line` citations into the Gwyddion
tree, and no source text.

Precedent: `docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/nfa/ATTRIBUTION.md`.
