# A11 drift register

| ID | Source behavior | PhenoTypic decision | Reason | Validation |
|---|---|---|---|---|
| PERS-D01 | GUDHI computes lower/sublevel filtrations | Superlevel passes `-image` and converts values back | Bright-ridge analysis needs descending intensity while public values remain interpretable | polarity fixture plus independent reduction |
| PERS-D02 | Pair representatives are Fortran-flat top-cell IDs | Expose `(row, column)` coordinates | Public image APIs use row/column coordinates | non-square fixture and ID conversion assertion |
| PERS-D03 | Essential pair APIs return only a birth coface | Append essential intervals with death-cell sentinel `(-1,-1)` and also expose `essential_cells` | Keep value arrays aligned while preserving explicit essential identity | single-cell polarity controls |
| PERS-D04 | GUDHI accepts broader dimensional inputs | Public function accepts exactly finite, nonempty real 2-D images | A11 is an image-analysis API with exactly beta-0/beta-1 output | planned invalid-input tests |
| PERS-D05 | GUDHI can compute arbitrary homology dimensions | Public tuple length is exactly two | A 2-D closed rectangle has relevant dimensions 0 and 1 | fixture and independent reduction |
| PERS-D06 | GUDHI returns source grouping/order | Preserve regular order and append essentials | Avoid inventing an interval sort while keeping aligned arrays | exact 3.13.0 fixture |
| PERS-D07 | GUDHI package is normally imported by callers | Import only inside a valid call and raise an actionable topology-extra error | Keep base package import-cheap | planned missing-dependency tests |
| PERS-D08 | Selected GUDHI corpus reports topology but does not reconstruct scalar images | Remove/defer persistence denoising and enhancer surfaces | No source-faithful reconstruction was established | scope review gate |
| PERS-D09 | The 2011 theory paper's image example uses pixels as vertices and 4-connectivity | Select GUDHI's top-dimensional-cell constructor, whose closed squares meet diagonally | Required for the matching `cofaces_of_persistence_pairs` API; conventions are not mixed | diagonal 8-connectivity control and independent boundary reduction |
