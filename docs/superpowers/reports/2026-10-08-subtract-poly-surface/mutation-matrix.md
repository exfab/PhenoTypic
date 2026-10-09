# SubtractPolySurface mutation matrix

Tree under test: commit `06283c5` (detached worktree, Slurm job 29656556, harness rc=0, baseline
127 passed). Harness and table: `docs/superpowers/plans/2026-10-08-subtract-poly-surface/run_mutations.py`,
`mutants.json`. Each mutant is one exact text replacement (anchor matched exactly once, checked for
all 17 before anything was touched); the file was restored from saved bytes after each mutant and its
sha256 re-verified (`restored_sha_ok` true for all 17). Raw output:
`.superpowers/sdd/plan/task-8-results-06283c5.json`.

Result: **16 of 17 killed; 1 equivalent** (`no_convergence_stop_lines`, argued and measured below).

| Mutant | Status | Failing tests | Killed by |
|---|---|---|---|
| `pivot_true_centre` | KILLED | 5 | `test_kernel_reproduces_the_oracle` (3 cases), `test_plane_through_the_operation`, `test_removes_tilt_about_gwyddions_pivot` |
| `plane_subtracts_constant` | KILLED | 7 | `test_constant_image_lands_at_its_documented_level`, `test_kernel_reproduces_the_oracle` (3 cases), `test_plane_through_the_operation`, `test_removes_tilt_about_gwyddions_pivot` |
| `line_no_avg` | KILLED | 25 | `test_constant_image_lands_at_its_documented_level`, `test_degree_zero_equals_the_row_shift_form`, `test_each_robust_row_matches_the_single_system_robust_fit`, `test_every_row_is_leveled_to_the_input_mean`, `test_kernel_reproduces_the_oracle` (12 cases), `test_line_levels_to_the_input_mean`, `test_removes_per_row_offsets_and_slopes_exactly`, `test_robust_clips_outliers_on_both_sides_of_the_median`, `test_robust_recovers_rows_that_are_mostly_background`, `test_rows_are_independent_across_block_boundaries` |
| `line_avg_after` | KILLED | 25 | `test_constant_image_lands_at_its_documented_level`, `test_degree_zero_equals_the_row_shift_form`, `test_each_robust_row_matches_the_single_system_robust_fit`, `test_every_row_is_leveled_to_the_input_mean`, `test_kernel_reproduces_the_oracle` (12 cases), `test_line_levels_to_the_input_mean`, `test_removes_per_row_offsets_and_slopes_exactly`, `test_robust_clips_outliers_on_both_sides_of_the_median`, `test_robust_recovers_rows_that_are_mostly_background`, `test_rows_are_independent_across_block_boundaries` |
| `swap_term_sets` | KILLED | 22 | `test_exact_data_returns_exact_coefficients`, `test_independent_is_the_tensor_product`, `test_independent_keeps_the_u3v3_term_and_total_degree_does_not`, `test_kernel_reproduces_the_oracle` (12 cases), `test_removes_its_own_form_exactly_and_lands_at_zero`, `test_term_counts`, `test_total_degree_keeps_p_plus_q_at_most_degree` |
| `normalize_by_n` | KILLED | 10 | `test_below_the_cap_every_pixel_is_used`, `test_divides_by_n_minus_one`, `test_endpoints_and_spacing`, `test_kernel_reproduces_the_oracle` (3 cases), `test_matches_dense_design_evaluation`, `test_plane_through_the_operation`, `test_removes_tilt_about_gwyddions_pivot` |
| `skip_column_transpose` | KILLED | 7 | `test_column_axis_is_row_axis_on_the_transpose`, `test_kernel_reproduces_the_oracle` (6 cases) |
| `one_sided_clip` | KILLED | 2 | `test_dark_colonies_are_clipped_too`, `test_each_robust_row_matches_the_single_system_robust_fit` |
| `one_sided_clip_lines` | KILLED | 2 | `test_each_robust_row_matches_the_single_system_robust_fit`, `test_robust_clips_outliers_on_both_sides_of_the_median` |
| `clipped_points_return` | KILLED | 2 | `test_clipped_points_never_return`, `test_each_robust_row_matches_the_single_system_robust_fit` |
| `clipped_points_return_lines` | KILLED | 1 | `test_each_robust_row_matches_the_single_system_robust_fit` |
| `no_convergence_stop` | KILLED | 1 | `test_max_iter_caps_the_rounds_and_convergence_stops_early` |
| `no_convergence_stop_lines` | SURVIVED | 0 | - |
| `no_rank_check` | KILLED | 1 | `test_rank_deficient_design_raises` |
| `no_min_inliers_guard` | KILLED | 1 | `test_never_fits_from_fewer_points_than_terms` |
| `no_min_inliers_guard_lines` | KILLED | 1 | `test_short_heavy_tailed_rows_never_fail` |
| `readmit_rejected_rows` | KILLED | 3 | `test_each_robust_row_matches_the_single_system_robust_fit`, `test_robust_recovers_rows_that_are_mostly_background`, `test_rows_are_independent_across_block_boundaries` |

(Tests live in `test_poly_surface_kernels.py`, `test_subtract_poly_surface.py` and
`test_subtract_poly_surface_gwyddion.py`; the Gwyddion fixture test is `test_kernel_reproduces_the_oracle`.)

## Load-bearing proof: `pivot_true_centre`

Replacing the plane pivot `(W/2, H/2)` with the pixel-index centre `((W-1)/2, (H-1)/2)` is killed by
the Gwyddion 2.71 fixture test (`test_kernel_reproduces_the_oracle[plane__A]`, `[plane__B]`,
`[plane__C]`) **and** by `TestPlane::test_removes_tilt_about_gwyddions_pivot`, plus
`TestOperationContract::test_plane_through_the_operation`. The fixture therefore fails when the one
convention the plane method exists to match is reintroduced wrongly: it is load-bearing. The
`normalize_by_n` mutant is killed by the same three plane cases.

## Tests added (before / after)

Run 1 (commit `d3b146f`, job 29655872): 14 killed, 3 survived: `one_sided_clip_lines`,
`clipped_points_return_lines`, `no_convergence_stop_lines`. All three sit in the per-line robust path
(`_robust_line_block`), whose existing tests only checked recovery quality on easy plates.

Added to `TestLevelLines` (commit `06283c5`):

- `test_robust_clips_outliers_on_both_sides_of_the_median[+1.0, -1.0]` kills `one_sided_clip_lines`
  (dark row defects must be clipped like bright colonies).
- `test_each_robust_row_matches_the_single_system_robust_fit` compares every row of a heavy-tailed
  (Student t, 2 dof) plate against `robust_least_squares` on that row alone, `atol=1e-8`. It kills
  `one_sided_clip_lines` and `clipped_points_return_lines` (a clipped point re-entering), and also
  `one_sided_clip`, `clipped_points_return` and `readmit_rejected_rows`.

Run 2 (commit `06283c5`, job 29656556): 16 killed, 1 equivalent.

## Equivalent mutant: `no_convergence_stop_lines`

Mutation: drop `& (count != mask.sum(axis=1))` from `refit`, so a row whose inlier mask did not change
this round stays active and is refit again.

Argument. `new = mask & (...)`, so `new` is a subset of `mask`; `count == mask.sum` therefore means
`new == mask` exactly. The refit builds `weight`, `gram` and `rhs` from that identical mask and solves
the same weighted normal equations that produced the current coefficients, so `coef` is unchanged up to
the difference between the first-round `lstsq` and the normal-equation solve. The next round recomputes
residual, median and scale from the same coefficients and mask and reaches the same decision again: the
extra rounds are a fixed point, a no-op on every reachable input. (The single-system analogue
`no_convergence_stop` is killed only because `RobustFit.rounds` exposes the round count; the per-line
path returns none.)

Measurement (scratchpad script, original vs mutant `level_lines`): 4800 random heavy-tailed cases
(up to 39 x 79, `line_order` 0-3, `clip_sigma` 1/1.5/3, `max_iter` 1/3/10/50) gave
`max |original - mutant| = 6.98e-12`, floating-point noise only. A test could separate them only with
a tolerance tighter than the solver difference, i.e. by asserting on rounding.

## Hygiene

Mutation ran only in detached worktrees at the commit under test; the main worktree's `src/` was never
mutated by the Slurm runs. The added tests were first checked in-tree against the three line mutants by
byte-restore (`restored_sha_ok` true). Detached worktrees are removed by `afterany` jobs.
