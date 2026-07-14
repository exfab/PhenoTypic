# A06 TrickTrack CA mutation matrix

Run `uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/cellular_automaton/run_mutations.py`.
The runner applies each change to a temporary module, loads only that mutant, runs the named test,
and fails if any mutant survives. The worktree is never edited.

| Mutant | Named killing test |
|---|---|
| Ordinary states update in place | `test_source_generated_fixture_matches_every_trace_and_result_field` |
| Scan transposed incoming neighbors | `test_evolution_scans_stored_outer_not_transposed_neighbors` |
| Retain the last equal neighbor instead of breaking at the first | `test_first_equal_scan_stops_at_first_stored_neighbor` |
| Run one extra ordinary round | `test_fixed_round_count_does_not_run_to_convergence` |
| Calculate all root flags from one global snapshot | `test_root_pass_is_immediate_and_preserves_supplied_order` |
| Change root threshold from `>=` to `>` | `test_root_threshold_is_inclusive` |
| Sort roots before the final pass | `test_root_pass_is_immediate_and_preserves_supplied_order` |
| Sort stored neighbors during DFS | `test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork` |
| Keep only the first fork | `test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork` |
| Continue beyond the exact path length | `test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork` |
| Promote source states from `uint8` to `int64` | `test_upper_minimum_preserves_uint8_state_and_full_exact_path` |
| Extract one extra cell per path | `test_dfs_preserves_neighbor_order_and_emits_every_exact_length_fork` |

The global-snapshot root mutant is the principal likely transcription bug. Its named test proves
the golden schedule fails when ordinary-round synchronization is incorrectly extended into the
source's special immediate root pass.
