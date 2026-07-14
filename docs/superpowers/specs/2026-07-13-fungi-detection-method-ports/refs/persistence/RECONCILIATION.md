# A11 source reconciliation

Production remains intentionally absent until independent G0 approval. This table reconciles the
frozen proposed contract and source-oracle harness line by line with the pinned executable source.

| Concern | GUDHI 3.13.0 evidence | Frozen A11 behavior | Status |
|---|---|---|---|
| License | `upstream/LICENSE:1-21` | Optional dependency; MIT attribution retained | exact |
| Version | `upstream/CMakeGUDHIVersion.txt:1-6` | Require tested oracle 3.13.0 | exact |
| Input construction | `upstream/src/python/gudhi/cubical_complex.py:29-105` | shaped `float64` array passed as `top_dimensional_cells` | exact plus validation |
| Flat order | `upstream/src/python/gudhi/cubical_complex.py:40-43,101-105` | interpret pair IDs in Fortran order | exact |
| Complex type | `upstream/src/python/gudhi/cubical_complex.py:88-105` | top-dimensional cells only; never vertices or periodic cells | exact |
| Paper convention | `wagner-chen-vucini-2011.txt:178-191` uses pixels as vertices and 4-connectivity | GUDHI top-dimensional-cell path has shared-corner 8-connectivity | explicit construction difference |
| Persistence field | `upstream/src/python/gudhi/cubical_complex.py:136-154` | field 11 | exact |
| Threshold | `upstream/src/python/gudhi/cubical_complex.py:146-150` | strictly `lifetime > min_persistence` | exact |
| Pair output | `upstream/src/python/gudhi/cubical_complex.py:173-236` | regular coface pairs plus essential birth cofaces | exact |
| Pair-cell ambiguity | `upstream/src/python/gudhi/cubical_complex.py:187-204` | exact cells are pinned drift data, not plateau invariants | exact |
| Homology dimensions | `upstream/src/python/gudhi/sklearn/cubical_persistence.py:101-124` | return beta-0 and beta-1 for 2-D inputs | narrowed public scope |
| Sublevel | `upstream/src/python/gudhi/cubical_complex.py:101-105,136-154` | pass image values unchanged | exact |
| Superlevel | no native superlevel switch | pass `-image`, then convert birth/death to original intensity | documented adaptation |
| Coordinates | source exposes Fortran-flat top-cell IDs | expose `(row, column)` and use `(-1,-1)` for essential death | documented adaptation |
| Essential values | source interval death is positive infinity in filtration coordinates | sublevel `+inf`; superlevel inverse is `-inf`; lifetime always `+inf` | documented adaptation |
| Ordering | `cofaces_of_persistence_pairs` returns grouped arrays | retain regular source order, append essential pairs | documented adaptation |
| Invalid values | source accepts a wider low-level domain | reject nonfinite/nonreal/non-2-D/empty inputs before optional import | safety guard |
| Reconstruction | no selected source/API line implements it | no denoiser or enhancer | scope correction |

The oracle-to-public conversion is executable in `generate_fixture.py`; the structurally different
oracle in the standalone logic script builds every cubical cell and reduces the boundary matrix
over `F_2` without importing GUDHI.
