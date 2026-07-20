# A11 required mutation matrix

These mutants were applied individually to a temporary import of the production source by
`test_required_mutants_are_killed`. The harness requires the named focused test to fail with pytest
exit code 1; syntax errors, collection errors, and failures in unrelated tests do not count as a
kill. The G2 run completed with all 15 mutation cases killed.

| Mutant | Killing test | G2 result |
|---|---|---|
| Remove/reverse the superlevel sign transform | `test_superlevel_uses_original_intensity_coordinates` | KILLED |
| Construct from vertices instead of top-dimensional cells | `test_top_cell_hole_and_pair_cells_match_fixture` | KILLED |
| Decode pair IDs in C order | `test_non_square_pair_ids_use_fortran_order` | KILLED |
| Swap row and column coordinates | `test_non_square_pair_ids_use_fortran_order` | KILLED |
| Swap beta-0 and beta-1 output dimensions | `test_top_cell_hole_and_pair_cells_match_fixture` | KILLED |
| Drop essential intervals | `test_essential_interval_sign_by_filtration` | KILLED |
| Make the strict threshold inclusive | `test_min_persistence_equality_is_excluded` | KILLED |
| Reverse the superlevel lifetime subtraction | `test_superlevel_uses_original_intensity_coordinates` | KILLED |
| Invent a canonical plateau representative | `test_plateau_cells_match_pinned_gudhi` | KILLED |
| Substitute the paper's vertex/4-connectivity convention | `test_diagonal_top_cells_share_a_vertex` | KILLED |
| Eagerly import GUDHI | `test_analysis_import_does_not_import_gudhi` | KILLED |
| Swallow missing GUDHI | `test_missing_gudhi_is_actionable` | KILLED |
| Remove the caller-input copy | `test_input_is_unchanged_and_result_does_not_alias_it` | KILLED |
| Return an image instead of persistence pairs | `test_result_contains_all_pair_arrays_with_frozen_shapes_and_dtypes` | KILLED |
| Add unsupported scalar reconstruction | `test_a11_has_no_operation_or_reconstruction_surface` | KILLED |

## Numerical comparison policy

The golden controls use exact array equality, not a fixed floating-point tolerance. GUDHI coface
IDs are integers; public values are direct `float64` indexing, sign reversal, subtraction, and
infinity insertion on the same pinned runtime. The controlled fixture values are exactly
representable binary values, so no rounding tolerance is needed. Cross-implementation validation
compares exact interval multisets and exact Betti counts in the standalone boundary-reduction
oracle. A future fixture with non-exact arithmetic must derive a scale-aware tolerance from the
specific operations before changing these assertions.
