# SubtractPolySurface mutation matrix

Tree under test: commit `ca0d02c` (detached worktree, Slurm job 29658151, baseline 129 passed). Harness and table:
`docs/superpowers/plans/2026-10-08-subtract-poly-surface/run_mutations.py`, `mutants.json`. Each mutant is
one exact text replacement (anchor matched exactly once, checked for all 18 before anything was touched);
the file is restored from saved bytes after each mutant and its sha256 re-verified (`restored_sha_ok`
true for all 18). A pytest return code other than 0 or 1 is `ERROR`, never a kill. Raw output:
`.superpowers/sdd/plan/task-8-results-ca0d02c.json`.

Result: **18 of 18 killed, 0 survivors, 0 equivalent.**

| Mutant | Status | Failing tests | Killed by |
|---|---|---|---|
| `pivot_true_centre` | KILLED | 5 | `test_kernel_reproduces_the_oracle` (3 cases), `test_plane_through_the_operation`, `test_removes_tilt_about_gwyddions_pivot` |
| `plane_subtracts_constant` | KILLED | 7 | `test_constant_image_lands_at_its_documented_level`, `test_kernel_reproduces_the_oracle` (3 cases), `test_plane_through_the_operation`, `test_removes_tilt_about_gwyddions_pivot` |
| `line_no_avg` | KILLED | 26 | `test_constant_image_lands_at_its_documented_level`, `test_degree_zero_equals_the_row_shift_form`, `test_each_robust_row_matches_the_single_system_robust_fit`, `test_every_row_is_leveled_to_the_input_mean`, `test_kernel_reproduces_the_oracle` (12 cases), `test_line_levels_to_the_input_mean`, `test_rejected_rows_match_the_single_system_fit`, `test_removes_per_row_offsets_and_slopes_exactly`, `test_robust_clips_outliers_on_both_sides_of_the_median`, `test_robust_recovers_rows_that_are_mostly_background`, `test_rows_are_independent_across_block_boundaries` |
| `line_avg_after` | KILLED | 26 | `test_constant_image_lands_at_its_documented_level`, `test_degree_zero_equals_the_row_shift_form`, `test_each_robust_row_matches_the_single_system_robust_fit`, `test_every_row_is_leveled_to_the_input_mean`, `test_kernel_reproduces_the_oracle` (12 cases), `test_line_levels_to_the_input_mean`, `test_rejected_rows_match_the_single_system_fit`, `test_removes_per_row_offsets_and_slopes_exactly`, `test_robust_clips_outliers_on_both_sides_of_the_median`, `test_robust_recovers_rows_that_are_mostly_background`, `test_rows_are_independent_across_block_boundaries` |
| `swap_term_sets` | KILLED | 22 | `test_exact_data_returns_exact_coefficients`, `test_independent_is_the_tensor_product`, `test_independent_keeps_the_u3v3_term_and_total_degree_does_not`, `test_kernel_reproduces_the_oracle` (12 cases), `test_removes_its_own_form_exactly_and_lands_at_zero`, `test_term_counts`, `test_total_degree_keeps_p_plus_q_at_most_degree` |
| `normalize_by_n` | KILLED | 10 | `test_below_the_cap_every_pixel_is_used`, `test_divides_by_n_minus_one`, `test_endpoints_and_spacing`, `test_kernel_reproduces_the_oracle` (3 cases), `test_matches_dense_design_evaluation`, `test_plane_through_the_operation`, `test_removes_tilt_about_gwyddions_pivot` |
| `skip_column_transpose` | KILLED | 7 | `test_column_axis_is_row_axis_on_the_transpose`, `test_kernel_reproduces_the_oracle` (6 cases) |
| `one_sided_clip` | KILLED | 3 | `test_dark_colonies_are_clipped_too`, `test_each_robust_row_matches_the_single_system_robust_fit`, `test_rejected_rows_match_the_single_system_fit` |
| `one_sided_clip_lines` | KILLED | 3 | `test_each_robust_row_matches_the_single_system_robust_fit`, `test_rejected_rows_match_the_single_system_fit`, `test_robust_clips_outliers_on_both_sides_of_the_median` |
| `clipped_points_return` | KILLED | 2 | `test_clipped_points_never_return`, `test_each_robust_row_matches_the_single_system_robust_fit` |
| `clipped_points_return_lines` | KILLED | 1 | `test_each_robust_row_matches_the_single_system_robust_fit` |
| `no_convergence_stop` | KILLED | 1 | `test_max_iter_caps_the_rounds_and_convergence_stops_early` |
| `no_convergence_stop_lines` | KILLED | 1 | `test_a_block_that_clips_nothing_performs_no_refit` |
| `no_rank_check` | KILLED | 1 | `test_rank_deficient_design_raises` |
| `no_min_inliers_guard` | KILLED | 2 | `test_never_fits_from_fewer_points_than_terms`, `test_rejected_rows_match_the_single_system_fit` |
| `no_min_inliers_guard_lines` | KILLED | 2 | `test_rejected_rows_match_the_single_system_fit`, `test_short_heavy_tailed_rows_never_fail` |
| `readmit_rejected_rows` | KILLED | 3 | `test_each_robust_row_matches_the_single_system_robust_fit`, `test_robust_recovers_rows_that_are_mostly_background`, `test_rows_are_independent_across_block_boundaries` |
| `revert_rejected_row_to_initial_fit` | KILLED | 1 | `test_rejected_rows_match_the_single_system_fit` |

(Tests live in `test_poly_surface_kernels.py`, `test_subtract_poly_surface.py` and
`test_subtract_poly_surface_gwyddion.py`; the Gwyddion fixture test is `test_kernel_reproduces_the_oracle`.)

Notes on individual rows:

- `line_avg_after` is indistinguishable from `line_no_avg` under `fit="lstsq"`: the order-0 Legendre column
  forces each row's residuals to sum to zero, so `(z - fitted).mean()` is about 0 and the mutant drops the
  restored mean. The two share their killers; they differ only under `fit="robust"`. Kept because it
  matches the spec's wording.
- `readmit_rejected_rows` is **broader than its name**: it keeps every non-refit row active and resets its
  mask to all-inlier, which includes rows that converged normally. Its kills come from that mask reset,
  not from D3. The D3 behaviour proper ("a rejected row keeps its last accepted fit") is pinned by the
  narrow mutant `revert_rejected_row_to_initial_fit` and
  `test_rejected_rows_match_the_single_system_fit`.
- `no_min_inliers_guard` (whole-surface guard) is killed by `test_never_fits_from_fewer_points_than_terms`;
  `no_min_inliers_guard_lines` (the per-line guard) is killed by `test_short_heavy_tailed_rows_never_fail`.
- Not covered by a mutant (whole-branch list): `center = median(r[kept])` taken over inliers, and the
  `scale == 0` stop (surface and per line). The single-system "keep previous fit on rejection" is asserted
  only as `kept.sum() >= n_terms` plus finiteness.

## Load-bearing proof: `pivot_true_centre`

Replacing the plane pivot `(W/2, H/2)` with the pixel-index centre `((W-1)/2, (H-1)/2)` is killed by
the Gwyddion 2.71 fixture test (`test_kernel_reproduces_the_oracle[plane__A]`, `[plane__B]`,
`[plane__C]`) **and** by `TestPlane::test_removes_tilt_about_gwyddions_pivot`, plus
`TestOperationContract::test_plane_through_the_operation`. The fixture therefore fails when the one
convention the plane method exists to match is reintroduced wrongly: it is load-bearing. The
`normalize_by_n` mutant is killed by the same three plane cases.

## Tests added (before / after)

Run 1 (commit `d3b146f`, job 29655872): 14 killed, 3 survived (`one_sided_clip_lines`,
`clipped_points_return_lines`, `no_convergence_stop_lines`), all in the per-line robust path.

Added to `TestLevelLines` (commit `06283c5`), run 2 (job 29656556): 16 killed, 1 survivor.

- `test_robust_clips_outliers_on_both_sides_of_the_median[+1.0, -1.0]` kills `one_sided_clip_lines`
  (`[+1.0]` is the control).
- `test_each_robust_row_matches_the_single_system_robust_fit` compares every row of a heavy-tailed
  (Student t, 2 dof) plate against `robust_least_squares` on that row alone, `atol=1e-8`; kills
  `one_sided_clip_lines`, `clipped_points_return_lines`, `one_sided_clip`, `clipped_points_return`.

Review round 1 (commit `ca0d02c`, this run):

- `test_rejected_rows_match_the_single_system_fit` runs the same per-row comparison on the D3-heavy input
  (Cauchy, seed 6, 200 x 6, `line_order=3`, `clip_sigma=1.0`; about 190 rows hit the D3 rejection). It is
  the only robust line input on which "keep the last accepted fit" is observable, and it kills
  `revert_rejected_row_to_initial_fit` (before: that mutant survived the finiteness-only test).
- `test_a_block_that_clips_nothing_performs_no_refit` spies on `np.linalg.solve`, the refit seam: bounded
  noise with a wide clip clips nothing in round 1, so no refit solve may happen; a plate with outliers must
  refit, which shows the spy sees the seam. It kills `no_convergence_stop_lines`.

## `no_convergence_stop_lines`: output-equivalent to rounding, killed on cost

The mutant drops `& (count != mask.sum(axis=1))` from `refit`, so a row whose inlier mask did not change
stays active and is refit every round up to `max_iter`. Its **output** is the original's up to rounding:
`new` is a subset of `mask`, so `count == mask.sum` means `new == mask`; the first extra refit replaces the
`lstsq` coefficients with the normal-equation solution on that same mask (this switch is the whole source
of the difference), and later rounds are a fixed point of that solve up to rounding. A clip decision lying
within rounding of `clip_sigma * scale` could in principle flip, none did.

Evidence (committed): `docs/superpowers/plans/2026-10-08-subtract-poly-surface/equivalence_no_convergence_stop_lines.py`
builds the mutant in memory and reports, over 4800 heavy-tailed cases (up to 39 x 79, `line_order` 0-3,
`clip_sigma` 1/1.5/3, `max_iter` 1/3/10/50): the mutant performed extra refit solves in 2074 cases, and
`max |original - mutant| = 6.976e-12`. So the code path is reached and the output differs by rounding only.

The cost is not equivalent: on a plate where round 1 clips nothing, the original makes 0 refit solves and the
mutant makes up to `max_iter` per converged block. The spy test above asserts exactly that, so the mutant is
**killed**, not excused.

## Hygiene

Slurm runs mutate only a detached worktree at the commit under test, removed by an `afterany` job. The
harness also installs a SIGTERM handler so the `finally` restore runs, bounds each mutant with a 900 s
pytest timeout, and writes the results JSON after every mutant.
