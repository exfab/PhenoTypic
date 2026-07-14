# A10 FilFinder adapter mutation evidence

The runner copies the private adapter to a temporary directory, applies one unique mutation at a
time, executes the named killing probe, and verifies the production SHA-256 afterward. It covers
the complete G0 required matrix plus warning task keys and the one-process constraint.

| Mutant | Semantic change | Named killing test | Result |
|---|---|---|---|
| FF-M01 | Skip the float32 ImageData precision seam | `test_private_source_copy_kills_direct_float64_threshold_mutant` | KILLED |
| FF-M02 | Make threshold comparison strict | `test_inclusive_threshold_and_nan_polarity` | KILLED |
| FF-M03 | Read gray instead of detect_mat | `test_source_arguments_are_copied_float64_and_unit_bearing` | KILLED |
| FF-M04 | Disable existing-mask mode | `test_exact_stage_graph_and_pool_shutdown` | KILLED |
| FF-M05 | Pass bare beam width | `test_source_arguments_are_copied_float64_and_unit_bearing` | KILLED |
| FF-M06 | Pass bare branch threshold | `test_source_arguments_are_copied_float64_and_unit_bearing` | KILLED |
| FF-M07 | Compensate upstream's one-pixel skeleton-threshold defect | `test_source_arguments_are_copied_float64_and_unit_bearing` | KILLED |
| FF-M08 | Swap relative-intensity and pruning fields | `test_source_arguments_are_copied_float64_and_unit_bearing` | KILLED |
| FF-M09 | Ignore the RNG seed | `test_source_arguments_are_copied_float64_and_unit_bearing` | KILLED |
| FF-M10 | Reuse a pool across applications | `test_fresh_source_object_and_pool_per_apply` | KILLED |
| FF-M11 | Select post-prune skeleton instead of longest path | `test_real_filfinder_matches_all_24_selected_oracle_outputs` | KILLED |
| FF-M12 | Label with 4-connectivity | `test_selected_rasters_use_eight_connectivity_and_row_major_labels` | KILLED |
| FF-M13 | Make objmask describe the threshold rather than selected raster | `test_selected_rasters_use_eight_connectivity_and_row_major_labels` | KILLED |
| FF-M14 | Modify detect_mat during application | `test_source_arguments_are_copied_float64_and_unit_bearing` | KILLED |
| FF-M15 | Import FilFinder eagerly | `test_module_import_is_optional_dependency_free` | KILLED |
| FF-M16 | Swallow the actionable missing-dependency error | `test_nonempty_mask_reports_missing_topology_extra` | KILLED |
| FF-M17 | Skip guaranteed pool shutdown | `test_pool_shutdown_is_guaranteed_after_failure` | KILLED |
| FF-M18 | Run downstream stages for mask output | `test_exact_stage_graph_and_pool_shutdown` | KILLED |
| FF-M19 | Suppress all warnings instead of one exact warning | `test_only_exact_supplied_mask_warning_is_suppressed` | KILLED |
| FF-M20 | Store worker warning records under the wrong task key | `test_real_process_pool_forwards_keyed_child_warning_to_parent` | KILLED |
| FF-M21 | Increase the owned pool above one process | `test_real_process_pool_forwards_keyed_child_warning_to_parent` | KILLED |

FF-M01 is the load-bearing historical mutant because the superseded float64-native oracle changed
threshold equality and two longest-path tie pixels. The private seam test and standalone logic
validator both fail if the float32 coercion is removed.
