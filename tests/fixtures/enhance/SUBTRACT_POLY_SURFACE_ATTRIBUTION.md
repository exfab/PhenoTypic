# SubtractPolySurface golden fixture — attribution

`subtract_poly_surface_gwyddion.npz` holds numeric outputs of Gwyddion 2.71
(Nečas & Klapetek, *Cent. Eur. J. Phys.* 10(1):181–188, 2012, doi:10.2478/s11534-011-0096-2),
GPL-2.0-or-later, computed by a harness that links Gwyddion's libgwyddion/libprocess on
deterministic inputs from `docs/superpowers/plans/2026-10-08-subtract-poly-surface/oracle_inputs.py`.

Only numbers are in this repository. No Gwyddion source, and not the harness (which calls
Gwyddion's API), is distributed, imported, linked, copied, or transcribed by PhenoTypic. The
production implementation (`src/phenotypic/enhance/_poly_surface_kernels.py`) was written by an
implementer who received `design.md` (a source-free behavioural contract), `drift-register.md`,
the logic-validation scripts, and this fixture — and did not open the Gwyddion source or user
guide. Provenance (tarball sha256, harness sha256, build) is in the sibling `.json`.
