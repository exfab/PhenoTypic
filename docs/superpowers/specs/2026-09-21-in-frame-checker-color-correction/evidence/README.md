# Evidence

The measurements the spec and `degree-and-patch-count.md` quote. Nothing here
is imported by the package or executed by the test suite — these are the
working files behind the numbers, kept so a reader can check a claim rather
than take it on faith, and so a future change can be compared against the
same substrate.

`patch_reduction_sweep.csv.gz` is gzipped (2.8 MB raw, 0.5 MB packed); read it
with `pandas.read_csv(..., compression="gzip")`.

## Which file backs which claim

| File | Backs |
|---|---|
| `patch_measurements.npz` | The substrate for everything below: 4 plate frames × 24 tiles, each with the fit representative and a seeded 1000-pixel sample. |
| `patch_importance_loo.csv`, `patch_importance_summary.csv` | §1 of the evidence doc — 768 leave-one-out fits, and cyan's 1.96 → 3.08 ΔE00 against under 0.44 for every other patch. |
| `patch_reduction_sweep.csv.gz`, `patch_reduction_summary.csv`, `patch_removal_permutations.csv` | §2 — 18 240 subset fits over 30 removal orders (seed 1729), the error-versus-patch-count curves, the 18-patch crossover and the miss budget. |
| `degree_effect_per_patch.csv` | §3 — the per-patch degree-2-versus-degree-3 difference, its Lab direction, and the bias-to-scatter ratio of 5.9×. |
| `degree_policy_between_frame.csv` | §3 — the five-policy table of apparent between-frame difference, including the 2.40 ΔE00 median for a mixed-degree batch. |
| `identity_placement_margin.csv` | Spec Stage D — placement margins for all 8 real half-cards under three hypothesis sets (0.476–0.606 for the 12 the code enumerates). |
| `identity_margin_threshold.csv` | Spec Stage D — the occlusion-injection calibration behind the 0.20 refuse / 0.25 warn gate. |
| `per_image_profile_grid.csv` | The per-image transfer grid: in-sample 1.95, cross-frame 3.03, within-session 2.09, across-session 3.98. **Holds only the 4×4 plate-frame block**; the chart-profile row is not in it. |
| `profile_fit_diagnostics.csv`, `geometric_median_tolerance_sweep.csv`, `estimator_comparison_per_tile.csv` | The estimator investigation that preceded this work: per-fit residuals and condition numbers, the `eps` sweep, and 120 tiles × 11 estimators. |
| `patch_count_findings.md`, `estimator_findings.md` | The memos written when those two investigations closed. |

## A caution on the accuracy numbers

`per_image_profile_grid.csv` and the memos describe the **earlier notebook
extraction**, not `CalibrateColorRpcc`. The operation currently measures
3.40 / 2.88 / 4.74 ΔE2000 in-sample on the same frames — worse than the
notebook's 1.95, with the gap diagnosed in Task 7 of the plan. Do not quote a
number from this folder as the operation's accuracy.
