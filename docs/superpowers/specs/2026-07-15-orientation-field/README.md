# Orientation-field analysis bundle

This folder preserves the complete July 14-15, 2026 orientation-zone analysis that
started from `LightDetectFungi_Workflow.ipynb` and was rerun with the
`branch-reconnection` TwoK filament detector output.

## Contents

- `design.md`: calculation definitions, interpretation, parameter choices, real-image
  results, limitations, and current recommendation.
- `scripts/`: every analysis and rendering script created during the session.
- `validation/`: independent NumPy/SciPy derivations of the load-bearing numeric
  invariants. These scripts do not import `phenotypic`.
- `../../logic_validation_scripts/2026-07-15-orientation-field/`: independent
  full-length ring-boundary and cumulative axial-change invariants.
- `artifacts/`: all generated figures, HTML views, CSV tables, JSON summaries, and
  session inspection images.
- `cache/`: the generated notebook composite, notebook object map, and
  branch-reconnection TwoK object map used for the real-image reruns.
- `source-manifest/changed-files.txt`: production and test files carried onto the
  `branch-reconnection` branch with this bundle.

## Source inputs

- Notebook used as the workflow reference:
  `/Users/alex/Projects/Neurospora/notebooks/LightDetectFungi_Workflow.ipynb`
- Original image path recorded by the scripts:
  `/Volumes/T9/exfab/UCR-010-I-D_Neurospora/data/denoised_media_subsets_FrameIdx10-12/xylan/d000273_300_001_2025-12-12_02-00-49_rgb.tiff`
- Real colonies emphasized in the final comparison:
  detector label 24 (`R3C4`) and detector label 36 (`R4C6`).

The original TIFF and notebook are external inputs and are not duplicated here. The
generated `.npy` cache files are included so expensive segmentation and detector stages
do not need to be repeated when those inputs are available.

## Final comparison artifacts

- `artifacts/twok_R3C4_tangential_methods_overlay_2x2.png`
- `artifacts/twok_R4C6_tangential_methods_overlay_2x2.png`
- `artifacts/twok_R3C4_ring_compounded_rotation_2x2.png`
- `artifacts/twok_R4C6_ring_compounded_rotation_2x2.png`
- `artifacts/twok_R3C4_equal_sector_axial_median_hsv_vs_rdbu.png`
- `artifacts/twok_R4C6_equal_sector_axial_median_hsv_vs_rdbu.png`
- `artifacts/twok_R3C4_equal_sector_axial_median_trimmed_hsv_vs_puor.png`
- `artifacts/twok_R4C6_equal_sector_axial_median_trimmed_hsv_vs_puor.png`
- `artifacts/twok_R3C4_ring_gate_diagnostic.png`
- `artifacts/twok_R3C4_ring_median_axial_change_2x2.png`
- `artifacts/twok_R4C6_ring_median_axial_change_2x2.png`
- `artifacts/twok_ring_median_axial_change_summary.csv`
- `artifacts/twok_ring_median_axial_change_profiles.csv`
- `artifacts/twok_R3C4_skeletonized_axial_change_2x3.png`
- `artifacts/twok_R4C6_skeletonized_axial_change_2x3.png`
- `artifacts/twok_skeletonized_axial_change_summary.csv`
- `artifacts/twok_skeletonized_axial_change_profiles.csv`
- `artifacts/twok_R3C4_point_matched_orientation_2x2.png`
- `artifacts/twok_R4C6_point_matched_orientation_2x2.png`
- `artifacts/twok_point_ring_crossings.csv`
- `artifacts/twok_point_matched_orientation_states.csv`
- `artifacts/twok_point_matched_orientation_summary.csv`
- `artifacts/twok_R3C4_ced_point_crossing_comparison_2x2.png`
- `artifacts/twok_R4C6_ced_point_crossing_comparison_2x2.png`
- `artifacts/twok_R3C4_ced_literal_crossing_trend_2x2.png`
- `artifacts/twok_R4C6_ced_literal_crossing_trend_2x2.png`
- `artifacts/twok_ced_literal_crossing_ring_profiles.csv`
- `artifacts/twok_ced_literal_crossing_summary.csv`
- `artifacts/twok_ced_literal_crossing_parameter_sweep.png`
- `artifacts/twok_ced_literal_crossing_parameter_sweep.csv`
- `artifacts/twok_ced_literal_crossing_parameter_sweep_aggregate.csv`
- `artifacts/twok_R3C4_literal_crossing_outward_metric_CED24.png`
- `artifacts/twok_R4C6_literal_crossing_outward_metric_CED24.png`
- `artifacts/twok_R3C4_literal_crossing_before_after_CED24.png`
- `artifacts/twok_R3C4_ced_point_crossing_comparison_CED24_2x2.png`
- `artifacts/twok_R4C6_ced_point_crossing_comparison_CED24_2x2.png`
- `artifacts/twok_tangential_methods_comparison.csv`
- `artifacts/twok_ring_compounded_rotation_summary.csv`
- `artifacts/twok_ring_compounded_rotation_profiles.csv`

## Reproduction notes

The scripts are preserved from the session. The ring-compounding dependency chain now
resolves its scripts, output directory, and included caches relative to this folder.
Other exploratory scripts can still contain absolute notebook or image paths. Before
rerunning from another machine, update those remaining input constants. Run project
commands with `uv`, for example:

```bash
uv run python docs/superpowers/specs/2026-07-15-orientation-field/validation/ring_compounded_rotation.py
```

The public orientation-zone schema was deliberately not expanded for the tangential or
ring-compounded prototypes. They remain diagnostic calculations pending further review.
The selected literal skeleton-ring crossing transform and its three independent plot
helpers are now public under `phenotypic.sdk_.orientation_fields`; no single phenotype
column has been selected from that profile.
The equal-sector axial mean/median ring-compounding prototype is intentionally different
from the other Sholl-style views: it starts outside the inferred inoculum but extends to
the first complete ring boundary beyond the farthest detected object pixel. It does not
use the symmetric-growth or sparse-zone radius as its outer limit.

## Helper relocation (2026-09-24)

The zone-measure helpers these scripts import were reorganised: shared numeric
primitives moved to `phenotypic.sdk_`, and helpers used only by
`MeasureOrientationZones` became private methods of that class. The arithmetic of
every moved function is unchanged; the orientation golden
(`test_orientation_zone_migration_golden.py`, 1e-10 tolerance) passes before and after.

**The scripts were deliberately not updated.** They are a record of the July session and
import from the old locations, so they run as written only against commit `fb3751f2`
(2026-09-24) or earlier:

```bash
git worktree add ../orientation-scripts fb3751f2
```

To port one to the current tree, apply this table:

| Old import | New location |
|---|---|
| `measure._measure_orientation_zones.signed_radial_relative_field` | `sdk_.orientation_fields.signed_radial_relative_field` |
| `…radial_ring_orientation_profile` | `sdk_.orientation_fields.radial_ring_orientation_profile` |
| `…radial_ring_sector_field` | `sdk_.orientation_fields.radial_ring_sector_field` |
| `…cumulative_ring_rotation_profile` | `sdk_.orientation_fields.cumulative_ring_rotation_profile` |
| `…long_range_ring_rotation_profile` | `sdk_.orientation_fields.long_range_ring_rotation_profile` |
| `…_FIBER_AXIS_OFFSET` | `sdk_.orientation_fields.FIBER_AXIS_OFFSET` |
| `…_RADIAL_RELATIVE_MIN_COHERENCE` | `sdk_.orientation_fields.RELIABLE_PIXEL_COHERENCE` |
| `…_RADIAL_RELATIVE_N_SECTORS` | `sdk_.orientation_fields.N_SECTORS` |
| `…zone_selector` | `MeasureOrientationZones._zone_selector` |
| `…aggregate_orientation` | `MeasureOrientationZones._aggregate_orientation` |
| `…_resultant_direction` | `MeasureOrientationZones._resultant_direction` |
| `…_BEND_SCALE_PRESETS` (module attribute) | `measure._orientation_zones._figures._BEND_SCALE_PRESETS` |
| `…radial_relative_field` | removed; use `np.abs` of the first value of `signed_radial_relative_field` |
| `util._orientation_field.orientation_field` | `sdk_.orientation_fields.orientation_field` |
| `util._matched_ring_rotation.*` | `sdk_.orientation_fields.*` (same names) |
| `util._nematic_bend.fiber_bend_field` | `sdk_.orientation_fields.fiber_bend_field` |
| `measure._zone_segmentation.distance_from_point` | `sdk_._radial_geometry.distance_from_point` |

`measure._zone_segmentation.compute_zone_segmentation` did not move.

Affected scripts (17): in `scripts/`, `neurospora_orientation_samples.py`,
`neurospora_radial_relative_samples.py`, `render_ced_point_crossing_comparison.py`,
`render_cumulative_orientation_samples.py`, `render_long_range_orientation_samples.py`,
`render_matched_ring_comparison.py`, `render_matched_rule_comparison.py`,
`render_point_matched_ring_orientation.py`, `render_ring_compounded_median_colormaps.py`,
`render_ring_compounded_rotation.py`, `render_ring_median_axial_change.py`,
`render_sigma_16_32_48.py`, `render_skeletonized_ring_median_axial_change.py`,
`render_tangential_method_comparison.py`, `scan_twok_long_scale_bend.py` and
`verify_implemented_radial_metrics_real.py`; and, outside this folder,
`docs/superpowers/artifacts/2026-09-01-zone-segmentation-improvement/make_extent_policy_comparison.py`.
The `validation/` scripts do not import `phenotypic` and are unaffected.
