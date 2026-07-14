# A11 required mutation matrix

These mutants are G1/G2 requirements after G0 approval. Each must be applied individually and
killed by the named focused test; a passing mutant blocks G3.

| Mutant | Intended killing test |
|---|---|
| Remove the superlevel sign transform | `test_superlevel_uses_original_intensity_coordinates` |
| Construct from vertices instead of top-dimensional cells | `test_top_cell_hole_and_pair_cells_match_fixture` |
| Decode pair IDs in C order | `test_non_square_pair_ids_use_fortran_order` |
| Swap row and column coordinates | `test_non_square_pair_ids_use_fortran_order` |
| Return only beta-0 or swap dimensions | `test_beta_one_hole_interval` |
| Drop or finite-clamp essential intervals | `test_essential_interval_sign_by_filtration` |
| Change strict `>` threshold to `>=` | `test_min_persistence_equality_is_excluded` |
| Swap birth/death or negate lifetime | `test_four_peak_lifetimes` |
| Canonicalize a plateau representative without contract approval | `test_plateau_cells_match_pinned_gudhi` |
| Use 4-connected foreground / 8-connected background semantics | `test_diagonal_top_cells_share_a_vertex` |
| Eagerly import GUDHI | `test_analysis_import_does_not_import_gudhi` |
| Swallow missing GUDHI and return empty arrays | `test_missing_gudhi_is_actionable` |
| Mutate the input array | `test_input_is_unchanged` |
| Return zeros, input, or an image-like array | `test_result_contains_all_pair_arrays` |
| Add scalar denoising or an enhancer wrapper | `test_a11_has_no_operation_or_reconstruction_surface` |
