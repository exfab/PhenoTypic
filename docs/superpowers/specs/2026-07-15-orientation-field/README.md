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
The equal-sector axial mean/median ring-compounding prototype is intentionally different
from the other Sholl-style views: it starts outside the inferred inoculum but extends to
the first complete ring boundary beyond the farthest detected object pixel. It does not
use the symmetric-growth or sparse-zone radius as its outer limit.
