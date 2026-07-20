# APP2 GWDT mutation-by-test matrix

Core mutants are injected one at a time by `run_mutations.py`; every row must report
`KILLED`. Seam mutants are owned by the serialized detector integration and are killed
by the focused tests named below.

| Mutant | Single conceptual change | Killing probe | Result |
|---|---|---|---|
| M01 background versus colony seeds | Invert the supplied seed mask | exact standard source distance | KILLED |
| M02 inverse versus raw intensity | Replace destination intensity with its reciprocal | standard 4-connectivity fixture | KILLED |
| M03 source versus destination intensity | Charge the current pixel rather than destination | load-bearing standard fixture | KILLED |
| M04 endpoint average inserted into GWDT | Average current and destination intensity | threshold/nonzero-seed fixture | KILLED |
| M05 frontier diagonal factor inserted | Multiply frontier destination by sqrt(offset) | initialization-diagonal fixture | KILLED |
| M06 ordinary diagonal factor removed | Replace ordinary step length with one | post-frontier-diagonal fixture | KILLED |
| M07 force 4-connectivity | Select four-neighbor table for requested 8 | standard 8-connectivity fixture | KILLED |
| M08 replace additive recurrence | Multiply accumulated and destination terms | standard 4-connectivity fixture | KILLED |
| M09 threshold `<` versus `<=` | Drop threshold-equality seed from generated source mask | threshold fixture | KILLED |
| M10 hidden normalization | Normalize image before GWDT | standard 4-connectivity fixture | KILLED |
| M11 invert/remove downstream lookup | Reverse or bypass fixed GI indices | exact source cost map | KILLED |
| M12 accept negative/nonfinite | Remove public value guards | invalid-input tests | KILLED |
| M13 change one-slice reduction | Treat `cnn_type=1` as diagonally connected | diagonal 4-connectivity fixture | KILLED |
| M14 initialize background to zero | Replace source input-valued seed distances | all-background fixture | KILLED |
| M15 no-background rejection/inf | Replace source float32 `1e20` sentinel behavior | no-background fixture | KILLED |
| M16 broken `else if` bounds scan | Recreate the active tree overload's coupled max/min update | strictly positive increasing cost control | KILLED |

## Serialized seam mutants

| Mutant | Killing test | Result |
|---|---|---|
| S01 raw cumulative GWDT | `test_full_image_adapter_applies_gi_lookup_not_raw_distance` | KILLED |
| S02 destination-only GI | `test_app2_axis_edges_use_endpoint_average_not_destination_cost` | KILLED |
| S03 exact square-root diagonal | `test_app2_diagonal_edges_use_pinned_source_factor` | KILLED |
| S04 skip full-image transform | `test_app2_cost_is_computed_once_on_full_image_before_tiling` | KILLED |
| S05 feed GI to legacy kernel | `test_tile_dispatch_keeps_app2_separate_from_legacy_dijkstra` | KILLED |
| S06 change disabled strategy | `test_tile_dispatch_keeps_app2_separate_from_legacy_dijkstra` | KILLED |
| S07 add source-style tree threshold gate | `test_app2_tree_allows_finite_gap_pixels_without_a_threshold_gate` | KILLED |
| S08 reverse equal-cost seed priority | `test_equal_cost_ownership_uses_first_row_major_boundary_seed` | KILLED |
| S09 restore exact Vaa3D neighbor priority | `test_equal_cost_predecessor_uses_fixed_detector_neighbor_order` | KILLED |
| S10 slice APP2 map from the global origin | `test_tiled_app2_uses_overlap_as_halo_and_exact_global_map_slices` | KILLED |
| S11 remove tile overlap halo | `test_tiled_app2_uses_overlap_as_halo_and_exact_global_map_slices` | KILLED |
| S12 let later tiles overwrite overlap ownership | `test_tiled_app2_uses_overlap_as_halo_and_exact_global_map_slices` | KILLED |
